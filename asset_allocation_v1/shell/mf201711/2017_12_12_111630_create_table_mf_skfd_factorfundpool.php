<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkfdFactorfundpool extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_skfd_factorfundpool', function($table) {
	    $table->increments('id');
	    $table->string('mf_id')->comment('多因子策略id,0*A股,1*A股基金,2*债券,3*债券基金,4*港股,5*港股基金,6*美股,7*美股基金');
	    $table->date('date')->default('0000-00-00');
	    $table->string('factor_name');
	    $table->integer('factor_end')->comment('基金所在端位：1-前端,0-后端');
	    $table->string('fd_code')->comment('基金代码');
	    $table->timestamps();
	});
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::drop('mf_skfd_factorfundpool');
    }
}
