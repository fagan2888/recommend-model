<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkfdPosition extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_skfd_position', function($table) {
	    $table->increments('id');
	    $table->string('fd_code')->comment('基金代码');
	    $table->string('sk_code')->comment('所持有股票代码');
	    $table->date('report_date')->default('0000-00-00')->comment('报告时间');
	    $table->date('publish_date')->default('0000-00-00')->comment('公布时间');
	    $table->decimal('position',7,4)->comment('持仓比例%');
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
        Schema::drop('mf_skfd_position');
    }
}
