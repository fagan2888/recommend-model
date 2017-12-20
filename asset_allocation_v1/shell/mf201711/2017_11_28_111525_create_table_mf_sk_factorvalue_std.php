<?php

use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTableMfSkFactorvalueStd extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('mf_sk_factorvalue_std', function($table) {
            $table->increments('id');
	    $table->string('mf_id')->comment('多因子策略id,0*A股,1*A股基金,2*债券,3*债券基金,4*港股,5*港股基金,6*美股,7*美股基金');
	    $table->date('periods_date');
            $table->string('sk_code');
            $table->string('factor_name');
            $table->decimal('factor_value_std',8,4)->comment('标准化因子值');
	    $table->integer('factor_position')->unsigned()->comment('分层所在层数，1为因子最前端');
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
        Schema::drop('mf_sk_factorvalue_std');
    }
}
